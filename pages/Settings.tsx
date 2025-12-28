
import React, { useState, useRef } from 'react';
import { useAppContext } from '../AppContext';
import type { StoreSettings, Service, Role, User, Permission } from '../types';
import { PlusIcon, TrashIcon, DownloadIcon, UploadIcon, UserGroupIcon, KeyIcon, WarningIcon, CheckIcon } from '../components/icons';
import Toast from '../components/Toast';
import { formatCurrency } from '../utils/formatters';
import { ALL_PERMISSIONS, groupPermissions } from '../utils/permissions';

interface TabProps {
    showToast: (message: string) => void;
}

const StoreDetailsTab: React.FC<TabProps> = ({ showToast }) => {
    const { storeSettings, updateSettings } = useAppContext();
    const [formData, setFormData] = useState(storeSettings);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        updateSettings(formData);
        showToast("مشخصات فروشگاه با موفقیت بروزرسانی شد.");
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-6 max-w-2xl mx-auto">
            <h3 className="text-xl font-bold text-slate-800 border-b pb-3 mb-4">مشخصات فروشگاه</h3>
            <div>
                <label htmlFor="storeName" className="block text-md font-semibold text-slate-700 mb-2">نام فروشگاه</label>
                <input id="storeName" name="storeName" value={formData.storeName} onChange={handleChange} className="w-full p-3 border rounded-lg form-input" />
            </div>
            <div>
                <label htmlFor="address" className="block text-md font-semibold text-slate-700 mb-2">آدرس</label>
                <input id="address" name="address" value={formData.address} onChange={handleChange} className="w-full p-3 border rounded-lg form-input" />
            </div>
            <div>
                <label htmlFor="phone" className="block text-md font-semibold text-slate-700 mb-2">شماره تماس</label>
                <input id="phone" name="phone" value={formData.phone} onChange={handleChange} className="w-full p-3 border rounded-lg form-input" />
            </div>
             <div>
                <label htmlFor="currencyName" className="block text-md font-semibold text-slate-700 mb-2">نام واحد پولی</label>
                <input id="currencyName" name="currencyName" value={formData.currencyName} onChange={handleChange} className="w-full p-3 border rounded-lg form-input" placeholder="مثال: افغانی" />
            </div>
            <div className="flex justify-end">
                <button type="submit" className="px-8 py-3 rounded-lg bg-blue-600 text-white btn-primary font-semibold">ذخیره تغییرات</button>
            </div>
        </form>
    );
};

const AlertsTab: React.FC<TabProps> = ({ showToast }) => {
    const { storeSettings, updateSettings } = useAppContext();
    const [formData, setFormData] = useState({
        lowStockThreshold: storeSettings.lowStockThreshold,
        expiryThresholdMonths: storeSettings.expiryThresholdMonths
    });

     const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: Number(value) }));
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        updateSettings({ ...storeSettings, ...formData });
        showToast("تنظیمات هشدارها با موفقیت بروزرسانی شد.");
    };

    return (
         <form onSubmit={handleSubmit} className="space-y-6 max-w-2xl mx-auto">
            <h3 className="text-xl font-bold text-slate-800 border-b pb-3 mb-4">مدیریت هشدارها</h3>
             <div className="p-4 bg-blue-50/70 border-l-4 border-blue-500 rounded-r-lg">
                <p className="text-blue-800">این تنظیمات به شما کمک می‌کنند تا قبل از اتمام موجودی یا تاریخ انقضای محصولات، از طریق داشبورد مطلع شوید.</p>
            </div>
            <div>
                <label htmlFor="lowStockThreshold" className="block text-md font-semibold text-slate-700 mb-2">آستانه کمبود موجودی</label>
                <input id="lowStockThreshold" name="lowStockThreshold" type="number" value={formData.lowStockThreshold} onChange={handleChange} className="w-full p-3 border rounded-lg form-input" />
                <p className="text-sm text-slate-500 mt-1">زمانی که موجودی یک کالا به این عدد یا کمتر برسد، هشدار داده خواهد شد.</p>
            </div>
             <div>
                <label htmlFor="expiryThresholdMonths" className="block text-md font-semibold text-slate-700 mb-2">بازه زمانی هشدار انقضا (به ماه)</label>
                <input id="expiryThresholdMonths" name="expiryThresholdMonths" type="number" value={formData.expiryThresholdMonths} onChange={handleChange} className="w-full p-3 border rounded-lg form-input" />
                 <p className="text-sm text-slate-500 mt-1">محصولاتی که تاریخ انقضای آن‌ها کمتر از این تعداد ماه آینده باشد، در داشبورد نمایش داده خواهند شد.</p>
            </div>
            <div className="flex justify-end">
                <button type="submit" className="px-8 py-3 rounded-lg bg-blue-600 text-white btn-primary font-semibold">ذخیره تغییرات</button>
            </div>
        </form>
    );
};


const ServicesTab: React.FC<TabProps> = ({ showToast }) => {
    const { services, addService, deleteService, storeSettings } = useAppContext();
    const [name, setName] = useState('');
    const [price, setPrice] = useState('');

    const handleAddService = () => {
        if (!name.trim() || !price || Number(price) <= 0) {
            showToast("لطفاً نام و قیمت معتبر برای خدمت وارد کنید.");
            return;
        }
        addService({ name: name.trim(), price: Number(price) });
        setName('');
        setPrice('');
        showToast("خدمت جدید با موفقیت اضافه شد.");
    };

    return (
        <div>
            <h3 className="text-xl font-bold text-slate-800 border-b pb-3 mb-4">تعریف خدمات</h3>
            <div className="flex gap-4 mb-6">
                <input value={name} onChange={e => setName(e.target.value)} placeholder="نام خدمت (مثال: جلد کردن کتاب)" className="flex-grow p-3 border rounded-lg form-input" />
                <input value={price} onChange={e => setPrice(e.target.value.replace(/[^0-9]/g, ''))} type="text" inputMode="numeric" placeholder={`قیمت (${storeSettings.currencyName})`} className="w-48 p-3 border rounded-lg form-input" />
                <button onClick={handleAddService} className="flex items-center bg-blue-600 text-white px-4 py-2 rounded-lg shadow-md btn-primary">
                    <PlusIcon className="w-5 h-5 ml-2" /> <span className="font-semibold">افزودن</span>
                </button>
            </div>
            <div className="space-y-3">
                {services.map(service => (
                    <div key={service.id} className="flex justify-between items-center p-4 bg-white/80 rounded-xl shadow-sm border">
                        <div>
                            <p className="font-bold text-slate-800 text-lg">{service.name}</p>
                            <p className="text-md text-blue-600 font-semibold">{formatCurrency(service.price, storeSettings)}</p>
                        </div>
                        <button onClick={() => deleteService(service.id)} className="text-red-500 hover:text-red-700 p-2 rounded-full hover:bg-red-100 transition-colors">
                           <TrashIcon className="w-6 h-6" />
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
};

const BackupRestoreTab: React.FC<TabProps> = ({ showToast }) => {
    const { 
        exportData, importData, cloudBackup, cloudRestore, 
        autoBackupEnabled, setAutoBackupEnabled, currentUser 
    } = useAppContext();
    
    const [isProcessing, setIsProcessing] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleImportClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            importData(file);
        }
        event.target.value = '';
    };

    const handleCloudBackup = async () => {
        setIsProcessing(true);
        await cloudBackup();
        setIsProcessing(false);
    };

    const handleCloudRestore = async () => {
        setIsProcessing(true);
        await cloudRestore();
        setIsProcessing(false);
    };

    return (
        <div className="space-y-10 max-w-4xl mx-auto">
            {/* Local Section */}
            <div className="bg-white/80 p-6 rounded-2xl shadow-lg border border-slate-200">
                <h3 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                    <DownloadIcon className="w-6 h-6 text-blue-600" />
                    پشتیبان‌گیری محلی (Local Backup)
                </h3>
                <p className="text-sm text-slate-600 mb-6 leading-relaxed">این گزینه یک فایل حاوی تمام اطلاعات فروشگاه شما در کامپیوتر دانلود می‌کند. برای بازیابی اطلاعات در آینده از همین فایل استفاده کنید.</p>
                
                <div className="flex flex-wrap gap-4">
                    <button onClick={exportData} className="flex items-center gap-2 px-6 py-3 rounded-xl bg-blue-600 text-white font-bold hover:bg-blue-700 transition-all shadow-md active:scale-95">
                        <DownloadIcon />
                        دانلود فایل پشتیبان
                    </button>
                    <button onClick={handleImportClick} className="flex items-center gap-2 px-6 py-3 rounded-xl bg-slate-200 text-slate-700 font-bold hover:bg-slate-300 transition-all shadow-md active:scale-95">
                        <UploadIcon />
                        بازیابی از فایل
                    </button>
                    <input type="file" accept=".json" ref={fileInputRef} onChange={handleFileChange} className="hidden" />
                </div>
            </div>

            {/* Cloud Section */}
            <div className="bg-gradient-to-br from-blue-50 to-white p-6 rounded-2xl shadow-lg border border-blue-200">
                <div className="flex justify-between items-start mb-4">
                    <h3 className="text-xl font-bold text-blue-800 flex items-center gap-2">
                        <UploadIcon className="w-6 h-6 text-blue-600" />
                        پشتیبان‌گیری آنلاین (Cloud Backup)
                    </h3>
                    <div className="flex items-center gap-2">
                         <span className={`text-xs px-2 py-1 rounded-full font-bold ${navigator.onLine ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                            {navigator.onLine ? 'متصل به ابر' : 'عدم اتصال به اینترنت'}
                        </span>
                    </div>
                </div>
                <p className="text-sm text-slate-600 mb-6 leading-relaxed">اطلاعات شما را در سرور مرکزی ذخیره می‌کند تا از هر دستگاهی به آن دسترسی داشته باشید. نسخه جدید جایگزین نسخه قبلی خواهد شد.</p>
                
                <div className="flex flex-wrap gap-4">
                    <button 
                        onClick={handleCloudBackup} 
                        disabled={isProcessing || !navigator.onLine}
                        className="flex items-center gap-2 px-6 py-3 rounded-xl bg-indigo-600 text-white font-bold hover:bg-indigo-700 transition-all shadow-md active:scale-95 disabled:bg-slate-400"
                    >
                        {isProcessing ? 'درحال انجام...' : 'ذخیره در ابر (Online Save)'}
                    </button>
                    <button 
                        onClick={handleCloudRestore}
                        disabled={isProcessing || !navigator.onLine}
                        className="flex items-center gap-2 px-6 py-3 rounded-xl bg-white border-2 border-indigo-600 text-indigo-600 font-bold hover:bg-indigo-50 transition-all shadow-md active:scale-95 disabled:border-slate-300 disabled:text-slate-400"
                    >
                        بازیابی از ابر (Online Restore)
                    </button>
                </div>

                <div className="mt-8 pt-6 border-t border-blue-200 flex flex-col md:flex-row items-center justify-between gap-4">
                    <div className="flex items-center gap-4">
                         <div className={`w-14 h-8 rounded-full p-1 cursor-pointer transition-all duration-300 ${autoBackupEnabled ? 'bg-green-500' : 'bg-slate-300'}`} onClick={() => setAutoBackupEnabled(!autoBackupEnabled)}>
                            <div className={`bg-white w-6 h-6 rounded-full shadow-md transition-all duration-300 ${autoBackupEnabled ? 'mr-6' : 'mr-0'}`}></div>
                        </div>
                        <div>
                            <p className="font-bold text-slate-800">پشتیبان‌گیری هوشمند ۲۴ ساعته</p>
                            <p className="text-xs text-slate-500">هر ۲۴ ساعت یک نسخه در ابر و یک نسخه در کامپیوتر ذخیره می‌شود.</p>
                        </div>
                    </div>
                    {autoBackupEnabled && (
                        <div className="flex items-center gap-2 text-green-600 font-bold text-sm bg-green-50 px-3 py-1.5 rounded-lg border border-green-200">
                            <CheckIcon className="w-4 h-4" />
                            فعال است
                        </div>
                    )}
                </div>
            </div>

             <div className="p-4 bg-red-50 border-r-4 border-red-500 rounded-l-lg flex gap-3">
                <WarningIcon className="w-6 h-6 text-red-600 shrink-0" />
                <p className="text-sm font-bold text-red-800 leading-relaxed">توجه امنیتی: بازیابی اطلاعات (چه از ابر و چه از فایل) تمام داده‌های فعلی برنامه را پاک کرده و نسخه پشتیبان را جایگزین می‌کند. لطفاً قبل از بازیابی، یک نسخه از اطلاعات فعلی خود ذخیره کنید.</p>
            </div>
        </div>
    );
};

const UsersAndRolesTab: React.FC<TabProps> = ({ showToast }) => {
    const { users, roles, addUser, updateUser, deleteUser, addRole, updateRole, deleteRole } = useAppContext();
    const [activeSubTab, setActiveSubTab] = useState<'users' | 'roles'>('users');
    
    // Role state
    const [editingRole, setEditingRole] = useState<Role | null>(null);
    const [roleName, setRoleName] = useState('');
    const [rolePermissions, setRolePermissions] = useState<Permission[]>([]);

    // User state
    const [editingUser, setEditingUser] = useState<User | null>(null);
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [userRoleId, setUserRoleId] = useState('');
    
    const groupedPermissions = groupPermissions(ALL_PERMISSIONS);
    
    const handleEditRole = (role: Role) => {
        setEditingRole(role);
        setRoleName(role.name);
        setRolePermissions(role.permissions);
    };

    const handleSaveRole = async () => {
        if (!roleName) { showToast("نام نقش نمی‌تواند خالی باشد."); return; }
        const result = await (editingRole 
            ? updateRole({ ...editingRole, name: roleName, permissions: rolePermissions })
            : addRole({ name: roleName, permissions: rolePermissions }));
        
        showToast(result.message);
        if(result.success) {
            setEditingRole(null);
            setRoleName('');
            setRolePermissions([]);
        }
    };
    
    const handlePermissionChange = (permissionId: string, checked: boolean) => {
        setRolePermissions(prev => checked ? [...prev, permissionId] : prev.filter(p => p !== permissionId));
    };

    const handleEditUser = (user: User) => {
        setEditingUser(user);
        setUsername(user.username);
        setUserRoleId(user.roleId);
        setPassword('');
    };
    
    const handleSaveUser = async () => {
        if(!username || !userRoleId || (!editingUser && !password)) {
            showToast("لطفا تمام فیلدها را پر کنید.");
            return;
        }
        const result = await (editingUser
            ? updateUser({ id: editingUser.id, username, roleId: userRoleId, password: password || undefined })
            : addUser({ username, password, roleId: userRoleId }));

        showToast(result.message);
        if(result.success) {
            setEditingUser(null);
            setUsername('');
            setPassword('');
            setUserRoleId('');
        }
    };
    

    return (
        <div>
            <div className="flex border-b mb-6">
                <button onClick={() => setActiveSubTab('users')} className={`p-3 font-semibold flex items-center gap-2 ${activeSubTab === 'users' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-slate-600'}`}><UserGroupIcon/> مدیریت کاربران</button>
                <button onClick={() => setActiveSubTab('roles')} className={`p-3 font-semibold flex items-center gap-2 ${activeSubTab === 'roles' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-slate-600'}`}><KeyIcon/> مدیریت نقش‌ها</button>
            </div>

            {activeSubTab === 'users' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                     <div>
                        <h4 className="text-lg font-bold mb-4">{editingUser ? 'ویرایش کاربر' : 'افزودن کاربر جدید'}</h4>
                        <div className="space-y-4 p-4 bg-white/70 rounded-lg border">
                            <input value={username} onChange={e => setUsername(e.target.value)} placeholder="نام کاربری" className="w-full p-2 border rounded form-input" />
                            <input value={password} onChange={e => setPassword(e.target.value)} type="password" placeholder={editingUser ? 'رمز عبور جدید (اختیاری)' : 'رمز عبور'} className="w-full p-2 border rounded form-input" />
                            <select value={userRoleId} onChange={e => setUserRoleId(e.target.value)} className="w-full p-2 border rounded bg-white form-input">
                                <option value="">-- انتخاب نقش --</option>
                                {roles.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
                            </select>
                            <div className="flex gap-2">
                                <button onClick={handleSaveUser} className="bg-blue-600 text-white px-4 py-2 rounded-lg btn-primary flex-grow">{editingUser ? 'بروزرسانی' : 'افزودن'}</button>
                                {editingUser && <button onClick={() => setEditingUser(null)} className="bg-gray-200 px-4 py-2 rounded-lg">لغو</button>}
                            </div>
                        </div>
                    </div>
                    <div>
                        <h4 className="text-lg font-bold mb-4">لیست کاربران</h4>
                        <ul className="space-y-2">
                            {users.map(user => (
                                <li key={user.id} className="flex justify-between items-center p-3 bg-white/70 rounded-lg border">
                                    <div>
                                        <p className="font-semibold">{user.username}</p>
                                        <p className="text-sm text-slate-600">{roles.find(r => r.id === user.roleId)?.name}</p>
                                    </div>
                                    <div className="flex gap-2">
                                        <button onClick={() => handleEditUser(user)} className="text-sm bg-slate-200 px-3 py-1 rounded-md">ویرایش</button>
                                        {user.username !== 'admin' && <button onClick={() => deleteUser(user.id)} className="text-sm bg-red-100 text-red-700 px-3 py-1 rounded-md">حذف</button>}
                                    </div>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            )}

            {activeSubTab === 'roles' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div>
                        <h4 className="text-lg font-bold mb-4">{editingRole ? 'ویرایش نقش' : 'افزودن نقش جدید'}</h4>
                        <div className="p-4 bg-white/70 rounded-lg border">
                            <input value={roleName} onChange={e => setRoleName(e.target.value)} placeholder="نام نقش" className="w-full p-2 border rounded mb-4 form-input" />
                            <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
                                {Object.entries(groupedPermissions).map(([group, permissions]) => (
                                    <div key={group}>
                                        <h5 className="font-bold border-b pb-1 mb-2">{group}</h5>
                                        {permissions.map(p => (
                                            <label key={p.id} className="flex items-center gap-2 p-1">
                                                <input type="checkbox" checked={rolePermissions.includes(p.id)} onChange={e => handlePermissionChange(p.id, e.target.checked)} className="h-4 w-4 rounded text-blue-600 focus:ring-blue-500"/>
                                                <span>{p.name}</span>
                                            </label>
                                        ))}
                                    </div>
                                ))}
                            </div>
                             <div className="flex gap-2 mt-4">
                                <button onClick={handleSaveRole} className="bg-blue-600 text-white px-4 py-2 rounded-lg btn-primary flex-grow">{editingRole ? 'بروزرسانی' : 'افزودن'}</button>
                                {editingRole && <button onClick={() => {setEditingRole(null); setRoleName(''); setRolePermissions([]);}} className="bg-gray-200 px-4 py-2 rounded-lg">لغو</button>}
                            </div>
                        </div>
                    </div>
                    <div>
                        <h4 className="text-lg font-bold mb-4">لیست نقش‌ها</h4>
                        <ul className="space-y-2">
                             {roles.map(role => (
                                <li key={role.id} className="flex justify-between items-center p-3 bg-white/70 rounded-lg border">
                                    <p className="font-semibold">{role.name}</p>
                                    <div className="flex gap-2">
                                        <button onClick={() => handleEditRole(role)} className="text-sm bg-slate-200 px-3 py-1 rounded-md">ویرایش</button>
                                        {role.name !== 'Admin' && <button onClick={() => deleteRole(role.id)} className="text-sm bg-red-100 text-red-700 px-3 py-1 rounded-md">حذف</button>}
                                    </div>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            )}
        </div>
    );
};


const Settings: React.FC = () => {
    const { hasPermission } = useAppContext();
    const [activeTab, setActiveTab] = useState('storeDetails');
    const [toast, setToast] = useState('');

    const showToast = (message: string) => {
        setToast(message);
        setTimeout(() => setToast(''), 3000);
    };

    const tabs = [
        { id: 'storeDetails', label: 'مشخصات فروشگاه', permission: 'settings:manage_store' },
        { id: 'alerts', label: 'هشدارها', permission: 'settings:manage_alerts' },
        { id: 'services', label: 'خدمات', permission: 'settings:manage_services' },
        { id: 'usersAndRoles', label: 'کاربران و نقش‌ها', permission: 'settings:manage_users' },
        { id: 'backup', label: 'پشتیبان‌گیری', permission: 'settings:manage_backup' },
    ];
    
    const accessibleTabs = tabs.filter(tab => hasPermission(tab.permission));

    const renderContent = () => {
        switch (activeTab) {
            case 'storeDetails': return <StoreDetailsTab showToast={showToast} />;
            case 'alerts': return <AlertsTab showToast={showToast} />;
            case 'services': return <ServicesTab showToast={showToast} />;
            case 'backup': return <BackupRestoreTab showToast={showToast} />;
            case 'usersAndRoles': return <UsersAndRolesTab showToast={showToast} />;
            default: return <StoreDetailsTab showToast={showToast} />;
        }
    };

    return (
        <div className="p-8">
            {toast && <Toast message={toast} onClose={() => setToast('')} />}
            <h1 className="mb-10">مرکز فرماندهی</h1>

            <div className="bg-white/60 backdrop-blur-xl rounded-2xl shadow-lg border border-gray-200/60">
                <div className="flex border-b border-gray-200/60 p-2 bg-white/40 rounded-t-2xl flex-wrap">
                    {accessibleTabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`py-3 px-6 font-bold text-lg rounded-lg transition-all duration-300 ${
                                activeTab === tab.id
                                    ? 'bg-white shadow-md text-blue-600'
                                    : 'text-slate-600 hover:bg-white/70 hover:text-blue-600'
                            }`}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>
                <div className="p-8">
                    {renderContent()}
                </div>
            </div>
        </div>
    );
};

export default Settings;
